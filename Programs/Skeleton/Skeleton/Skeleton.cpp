//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2018. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Tuska Jozsef Csongor
// Neptun : LAU37R
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
//=============================================================================================
// Computer Graphics Sample Program: GPU ray casting
//=============================================================================================
//=============================================================================================
// Path tracing program
//=============================================================================================
//=============================================================================================
// Computer Graphics Sample Program: GPU ray casting
//=============================================================================================
//=============================================================================================
// Computer Graphics Sample Program: GPU ray casting
//=============================================================================================
#include "framework.h"

const char *vertexSource = R"(
	#version 450
    precision highp float;

	uniform vec3 wLookAt, wRight, wUp;          // pos of eye

	layout(location = 0) in vec2 cCamWindowVertex;	// Attrib Array 0
	out vec3 p;

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";

const char *fragmentSource = R"(
	#version 450
    precision highp float;

	struct Material {
		vec3 ka, kd, ks;
		float  shininess;
		vec3 F0;
		int rough, reflective;
	};

	struct Light {
		vec3 direction;
		vec3 Le, La;
	};

	struct Sphere {
		vec3 center, scale;
		float radius;
		int mat;
	};

	struct Mirror{
		vec3 p1;
		vec3 p2;
		vec3 p3;
		vec3 p4;
		int mat;
	};

	struct Hit {
		float t;
		vec3 position, normal;
		int mat;	// material index
	};

	struct Ray {
		vec3 start, dir;
	};

	const int nMaxObjects = 3;
	const int nMaxMirrors = 150;

	uniform vec3 wEye; 
	uniform Light light;     
	uniform Material materials[10];  // diffuse, specular, ambient ref
	uniform int nObjects;
	uniform int nMirrors;
	uniform Mirror mirrors[nMaxMirrors];
	uniform Sphere objects[nMaxObjects];

	in  vec3 p;					// point on camera window corresponding to the pixel
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	Hit intersect(const Sphere object, const Ray ray) {
		Hit hit;
		vec3 scale=object.scale;
		hit.t = -1;
		vec3 dist = ray.start*scale - object.center *scale;
		vec3 dir = ray.dir * scale;
		
		float a = dot(dir, dir);
		float b = dot(dist, dir) * 2.0;
		float c = dot(dist, dist) - object.radius * object.radius;
		float discr = b * b - 4.0 * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2.0 / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0 / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start*scale + ray.dir * hit.t*scale;
		hit.normal = normalize(hit.position - object.center*scale);
		hit.mat = object.mat;
		return hit;
	}

	Hit intersect(const Mirror mirror, const Ray ray){
		Hit hit;
		hit.t = -1;

		vec3 r1 = mirror.p1;
		vec3 r2 = mirror.p2;
		vec3 r3 = mirror.p3;
		vec3 r4 = mirror.p4;
		
		vec3 n = normalize(cross((r2-r1), (r3-r1)));

		float t = dot(r1-ray.start, n)/dot(ray.dir, n);

		if(t<0) return hit;

		vec3 p = ray.start + ray.dir * t;
		
		float product1 = dot(cross(r2-r1, p-r1), n); if(product1 < 0) return hit;
		float product2 = dot(cross(r3-r2, p-r2), n); if(product2 < 0) return hit;
		float product3 = dot(cross(r4-r3, p-r3), n); if(product3 < 0) return hit;
		float product4 = dot(cross(r1-r4, p-r4), n); if(product4 < 0) return hit;

		hit.position=p;
		hit.t=t;
		hit.normal=n;
		hit.mat= mirror.mat;
	
		return hit;
}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;

		for (int o = 0; o < nObjects; o++) {
			Hit hit = intersect(objects[o], ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}


		for (int o = 0; o < nMirrors; o++) {
			Hit hit = intersect(mirrors[o], ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}

		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);

		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		bool shadow = false;
		for (int o = 0; o < nObjects; o++) if (intersect(objects[o], ray).t > 0) shadow = true;  //  hit.t < 0 if no intersection
		//for (int o = 0; o < nMirrors; o++) if (intersect(mirrors[o], ray).t > 0) shadow = true; //  hit.t < 0 if no intersection
		return shadow;
	}

	vec3 Fresnel(vec3 F0, float cosTheta) { 
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}

	const float epsilon = 0.0001f;
	const int maxdepth = 40;

	vec3 trace(Ray ray) {
		vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = vec3(0, 0, 0);
		for(int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) return weight * light.La;
			if (materials[hit.mat].rough == 1) {
				outRadiance += weight * materials[hit.mat].ka * light.La;
				Ray shadowRay;
				shadowRay.start = hit.position + hit.normal * epsilon;
				shadowRay.dir = light.direction;
				float cosTheta = dot(hit.normal, light.direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance += weight * light.Le * materials[hit.mat].kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light.direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance += weight * light.Le * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
				}
			}

			if (materials[hit.mat].reflective == 1) {
				weight *= Fresnel(materials[hit.mat].F0, dot(-ray.dir, hit.normal));
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
			} else return outRadiance;
		}
	}

	void main() {
		Ray ray;
		ray.start = wEye; 
		ray.dir = normalize(p - wEye);
		fragmentColor = vec4(trace(ray), 1); 
	}
)";

class Material {
protected:
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	bool rough, reflective;
public:
	Material RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
	Material SmoothMaterial(vec3 _F0) {
		F0 = _F0;
		rough = false;
		reflective = true;
	}
	void SetUniform(unsigned int shaderProg, int mat) {
		char buffer[256];
		sprintf(buffer, "materials[%d].ka", mat);
		ka.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].kd", mat);
		kd.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].ks", mat);
		ks.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].shininess", mat);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, shininess); else printf("uniform material.shininess cannot be set\n");
		sprintf(buffer, "materials[%d].F0", mat);
		F0.SetUniform(shaderProg, buffer);

		sprintf(buffer, "materials[%d].rough", mat);
		location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1i(location, rough ? 1 : 0); else printf("uniform material.rough cannot be set\n");
		sprintf(buffer, "materials[%d].reflective", mat);
		location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1i(location, reflective ? 1 : 0); else printf("uniform material.reflective cannot be set\n");
	}
};

class RoughMaterial : public Material {
public:
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
};

class SmoothMaterial : public Material {
public:
	SmoothMaterial(vec3 _F0) {
		F0 = _F0;
		rough = false;
		reflective = true;
	}
};

struct Sphere {
	vec3 center, scale;
	float radius;
	int mat;

	Sphere(const vec3& _center, const vec3& _scale, float _radius,const int _mat) { center = _center; scale = _scale; radius = _radius; mat = _mat; }
	void SetUniform(unsigned int shaderProg, int o) {
		char buffer[256];
		sprintf(buffer, "objects[%d].center", o);
		center.SetUniform(shaderProg, buffer);

		sprintf(buffer, "objects[%d].scale", o);
		scale.SetUniform(shaderProg, buffer);

		sprintf(buffer, "objects[%d].radius", o);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, radius); else printf("uniform %s cannot be set\n", buffer);

		sprintf(buffer, "objects[%d].mat", o);
		location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1i(location, mat); else printf("uniform %s cannot be set\n", buffer);
	}
};

struct Mirror {
	vec3 p1, p2, p3, p4;
	int mat;

	Mirror(const vec3& _p1,const vec3& _p2, const vec3& _p3, const vec3& _p4, int _mat) {
		p1 = _p1;
		p2 = _p2;
		p3 = _p3;
		p4 = _p4;
		mat = _mat;
	}

	void SetUniform(unsigned int shaderProg, int o) {
		char buffer[512];
		sprintf(buffer, "mirrors[%d].p1", o);
		p1.SetUniform(shaderProg, buffer);

		sprintf(buffer, "mirrors[%d].p2", o);
		p2.SetUniform(shaderProg, buffer);

		sprintf(buffer, "mirrors[%d].p3", o);
		p3.SetUniform(shaderProg, buffer);

		sprintf(buffer, "mirrors[%d].p4", o);
		p4.SetUniform(shaderProg, buffer);

		sprintf(buffer, "mirrors[%d].mat", o);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1i(location, mat); else printf("uniform %s cannot be set\n", buffer);
	}
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, double _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tan(fov / 2);
		up = normalize(cross(w, right)) * f * tan(fov / 2);
	}
	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		set(eye, lookat, up, fov);
	}
	void SetUniform(unsigned int shaderProg) {
		eye.SetUniform(shaderProg, "wEye");
		lookat.SetUniform(shaderProg, "wLookAt");
		right.SetUniform(shaderProg, "wRight");
		up.SetUniform(shaderProg, "wUp");
	}

	void Zoom() {
		fov *= 0.5;
	}

	void Move(vec3 dir) {
		eye.x += dir.x;
		eye.y += dir.y;
		eye.z += dir.z;

	}
};

struct Light {
	vec3 direction;
	vec3 Le, La;
	Light(vec3 _direction, vec3 _Le, vec3 _La) {
		direction = normalize(_direction);
		Le = _Le; La = _La;
	}
	void SetUniform(unsigned int shaderProg) {
		La.SetUniform(shaderProg, "light.La");
		Le.SetUniform(shaderProg, "light.Le");
		direction.SetUniform(shaderProg, "light.direction");
	}
};


float rnd() { return (float)rand() / RAND_MAX; }

class Scene {
	std::vector<Sphere *> objects;
	std::vector<Light *> lights;
	std::vector<Material *> materials;
	std::vector<Mirror *> mirrors;
	int nMirrors = 3;
	bool dir = false;

	Camera camera;
public:
	void build() {
		vec3 eye = vec3(0, 0, 2);
		vec3 vup = vec3(0, 1, 0);
		vec3 lookat = vec3(0, 0, 0);
		float fov = 120 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		lights.push_back(new Light(vec3(0, 0, 0.5f), vec3(3, 3, 3), vec3(0.7, 0.7, 0.7)));
		vec3 kd(1.0f, 0, 0), ks(20, 20, 20);
		vec3 kd2(0, 1, 0);
		vec3 kd3(0, 0, 1);

		objects.push_back(new Sphere(vec3(0, 0, -2), vec3(0.1, 0.3, 0.4), 0.01, 0));
		objects.push_back(new Sphere(vec3(0.3, 0, -2), vec3(0.1, 0.3, 0.4), 0.01, 1));
		objects.push_back(new Sphere(vec3(-0.3, 0, -2), vec3(0.1, 0.3, 0.4), 0.01, 2));

		BuildMirrors();

		materials.push_back(new RoughMaterial(kd, ks, 50));
		materials.push_back(new RoughMaterial(kd2, ks, 50));
		materials.push_back(new RoughMaterial(kd3, ks, 50));
		materials.push_back(new SmoothMaterial(vec3(0.9381f, 0.8464f, 0.3915f)));
	}
	void SetUniform(unsigned int shaderProg) {
		int location = glGetUniformLocation(shaderProg, "nObjects");
		if (location >= 0) glUniform1i(location, objects.size()); else printf("uniform nObjects cannot be set\n");

		location = glGetUniformLocation(shaderProg, "nMirrors");
		if (location >= 0) glUniform1i(location, mirrors.size()); else printf("uniform nMirrors cannot be set\n");

		for (int m = 0; m < mirrors.size(); m++) mirrors[m]->SetUniform(shaderProg, m);
		for (int o = 0; o < objects.size(); o++) objects[o]->SetUniform(shaderProg, o);
	

		lights[0]->SetUniform(shaderProg);
		camera.SetUniform(shaderProg);
		for (int mat = 0; mat < materials.size(); mat++) materials[mat]->SetUniform(shaderProg, mat);
	}
	void Animate(float dt) { camera.Animate(dt); }

	void MoveLights(vec3 dir) {
		lights[0]->direction.x += dir.x; lights[0]->direction.y += dir.y; lights[0]->direction.z += dir.z;}

	void BuildMirrors() {
		mirrors.clear();
		for (int i = 0; i < nMirrors; i++) {
			float angle = (float)i / nMirrors * 2 * M_PI;
			float angle2 = (float)(i + 1) / nMirrors * 2 * M_PI;
			mirrors.push_back(new Mirror(vec3(cosf(angle), sinf(angle), 5), vec3(cosf(angle), sinf(angle), -2), vec3(cosf(angle2), sinf(angle2), -2), vec3(cosf(angle2), sinf(angle2), 5), 3));
		}
	}

	void IncreaseMirrors(int i) {
		if(nMirrors<150)
		nMirrors = nMirrors + i;
	}

	void MoveEllipsoids() {
		for (int i = 0; i < objects.size(); i++) {
			/*if (sqrtf((0 - objects[i]->center.x)*(0 - objects[i]->center.x)* (0 - objects[i]->center.y)*(0 - objects[i]->center.y) *(0 - objects[i]->center.z)*(0 - objects[i]->center.z)) > 0.5f)
				dir = !dir;

			int random = rand();

			if(dir){

				if(random %3 ==0)
					objects[i]->center.x += 0.01;

				if (random % 3 == 1)
					objects[i]->center.y += 0.01;

				if (random % 3 == 2)
					objects[i]->center.z += 0.01;

			}

			else {
				if (random % 3 == 0)
					objects[i]->center.x -= 0.01;

				if (random % 3 == 1)
					objects[i]->center.y -= 0.01;

				if (random % 3 == 2)
					objects[i]->center.z -= 0.01;
			}*/


			vec3 p1, p2, p3;
			p1.x = (rnd() * 2 - 1) *0.6;
			p1.y = (rnd() * 2 - 1) *0.6;
			p1.z = 0;
			
			vec3 dir1 = p1 - objects[i]->center;

			while (length(dir1)>0.01) {
				objects[i]->center = objects[i]->center + dir1 * 0.00001f;
				dir1 = p1 - objects[i]->center;
				glutPostRedisplay();
			}

		}
	}

};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
public:
	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		
		glGenBuffers(1, &vbo);	

		glBindBuffer(GL_ARRAY_BUFFER, vbo); 
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);   
	}

	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	
	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad.Create();

	gpuProgram.Create(vertexSource, fragmentSource, "fragmentColor");
	gpuProgram.Use();
}

void onDisplay() {
	static int nFrames = 0;
	nFrames++;
	static long tStart = glutGet(GLUT_ELAPSED_TIME);
	long tEnd = glutGet(GLUT_ELAPSED_TIME);

	glClearColor(0.1f, 0, 0.8f, 1.0f);							
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
	scene.SetUniform(gpuProgram.getId());
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();									
}

void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key)
	{
	case 'a':  scene.IncreaseMirrors(1);
		scene.BuildMirrors();
		break;

	case 'd': scene.IncreaseMirrors(-1);
		scene.BuildMirrors();
		break;

	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	static float count = 0;
	count++;
	if (count == 5000) {
		scene.MoveEllipsoids();
		count = 0;
	}
	
}